subroutine setprob()

    use breach_module, only: setup_breaching
!    use breach_module, only: mu
!    use breach_module, only: lat0
!    use breach_module, only: lat1
!    use breach_module, only: lon0
!    use breach_module, only: lon1
!    use breach_module, only: sigma
!    use breach_module, only: breach_trigger
!    use breach_module, only: time_ratio
!    use breach_module, only: start_time
!    use breach_module, only: end_time

    implicit none
!    real(kind=8), intent(inout) :: center, west, east, north, south, t0, t1, trig, var, tr
    call setup_breaching('breach.data')

end subroutine setprob
